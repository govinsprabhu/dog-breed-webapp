import { Injectable } from '@angular/core'
import { Http, RequestOptions, Headers } from '@angular/http'
import { environment } from '../../environments/environment'
import { map } from 'rxjs/operators';




@Injectable()
export class DogBreedService {
    private baseUrlPath: any;


    constructor(private _http: Http) {
        this.baseUrlPath = environment.url
    }

    headers = new Headers({ 'Content-Type': 'application/json' });
    options = new RequestOptions({ headers: this.headers });

    getTheBreed(fileData) {
        console.log('inside the getTheBreed  ',fileData);
        let url = environment.url;
        let headers = new Headers();
        headers.append('Accept', 'application/json');
        let options = new RequestOptions({ headers: headers });
        return this._http.post(url +'/upload',
            fileData
            , options).pipe(map((res: any) => res.json()));
    }


}